// Render the console's HTML template with React. Bindings are property paths,
// and nested templates provide conditional sections and repeated rows.
(function () {
  'use strict';

  const root = document.getElementById('leopt-root');
  if (!window.React || !window.ReactDOM || !window.LeoptConsole) {
    root.textContent = 'The console could not load. Check your connection and reload.';
    root.style.cssText = 'color:#aab0ba;padding:24px;font-family:system-ui';
    return;
  }

  const h = React.createElement;
  const attributeNames = {
    class: 'className', for: 'htmlFor', spellcheck: 'spellCheck',
    viewbox: 'viewBox', 'stroke-width': 'strokeWidth',
    'stroke-dasharray': 'strokeDasharray',
    onclick: 'onClick', onchange: 'onChange', oninput: 'onInput',
    ondragover: 'onDragOver', ondragleave: 'onDragLeave', ondrop: 'onDrop',
  };

  function lookup(values, path) {
    return path.trim().split('.').reduce((value, key) => value?.[key], values);
  }

  function binding(source) {
    const whole = source.match(/^\s*\{\{\s*([^{}]+?)\s*\}\}\s*$/);
    if (whole) return values => lookup(values, whole[1]);
    if (!source.includes('{{')) return () => source;
    const parts = source.split(/\{\{([^{}]+)\}\}/g);
    return values => parts.map((part, i) => i % 2 ? lookup(values, part) ?? '' : part).join('');
  }

  function styleObject(css) {
    if (!css || typeof css === 'object') return css;
    const style = {};
    for (const declaration of css.split(';')) {
      const colon = declaration.indexOf(':');
      if (colon < 0) continue;
      const name = declaration.slice(0, colon).trim();
      const key = name.startsWith('--') ? name : name.replace(/-([a-z])/g, (_, c) => c.toUpperCase());
      style[key] = declaration.slice(colon + 1).trim();
    }
    return style;
  }

  function compileChildren(parent) {
    return [...parent.childNodes].map(compile).filter(Boolean);
  }

  function childrenAt(children, values) {
    return children.map((render, key) => render(values, key));
  }

  function compile(node) {
    if (node.nodeType === Node.TEXT_NODE) {
      const read = binding(node.textContent);
      return values => read(values) ?? '';
    }
    if (node.nodeType !== Node.ELEMENT_NODE) return null;

    if (node.localName === 'template') {
      const children = compileChildren(node.content);
      if (node.hasAttribute('data-if')) {
        const condition = binding(node.getAttribute('data-if'));
        return (values, key) => condition(values)
          ? h(React.Fragment, { key }, childrenAt(children, values)) : null;
      }
      if (node.hasAttribute('data-each')) {
        const read = binding(node.getAttribute('data-each'));
        const alias = node.getAttribute('data-as') || 'item';
        return (values, key) => h(React.Fragment, { key },
          (read(values) || []).map((item, i) => h(React.Fragment, { key: i },
            childrenAt(children, { ...values, [alias]: item }))));
      }
      throw new Error('Console template requires data-if or data-each');
    }

    const component = node.getAttribute('data-component');
    const children = compileChildren(node);
    const attributes = [...node.attributes]
      .filter(({ name }) => name !== 'data-component')
      .map(({ name, value }) => {
        const key = attributeNames[name] || name;
        const read = binding(value);
        if (key !== 'style') return [key, read];
        if (!value.includes('{{')) {
          const style = styleObject(value);
          return [key, () => style];
        }
        return [key, values => styleObject(read(values))];
      });

    return (values, key) => {
      const props = { key };
      for (const [name, read] of attributes) {
        props[name] = read(values);
        if (name === 'value' && props[name] == null) props[name] = '';
      }
      const content = component
        ? [h(window[component], { key: 'component' })]
        : childrenAt(children, values);
      return content.length ? h(node.localName, props, content) : h(node.localName, props);
    };
  }

  const template = compileChildren(document.getElementById('console-template').content);
  window.renderLeoptTemplate = values => childrenAt(template, values);
  ReactDOM.createRoot(root).render(h(window.LeoptConsole));
})();
